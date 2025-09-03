import pandas as pd
import geopandas as gpd

def overlap(parcels: gpd.GeoDataFrame,zones: gpd.GeoDataFrame,joined_gdf: gpd.GeoDataFrame, 
            apn: str = "APN", zone: str = "ZONING", zone_out: str = "ZONING_final") -> gpd.GeoDataFrame:
    """
    Resolve zoning overlaps to a single zone per parcel.

    Parameters:
        parcels : GeoDataFrame
            Parcel polygons; must contain `{apn}` and a projected CRS.
        zones : GeoDataFrame
            Zoning polygons; must contain `{zone}` and same CRS.
        joined_gdf : GeoDataFrame
            Parcel–zone intersection from base join, with columns `{apn}`, `{zone}`, and intersection `geometry`.
        apn, zone, zone_out: str
            Field names.

    Returns
        GeoDataFrame
            Copy of `parcels` with `{zone_out}` assigned to the zone with the
            largest overlap area per parcel.
    """

    # Singles
    # isolate 1-to-1 APN-Zone pairs from joined_gdf
    zones_per_apn = joined_gdf.groupby(apn)[zone].nunique(dropna=True)
    single_ids = zones_per_apn[zones_per_apn == 1].index
    # mapping APN -> zone for single ids
    single_map = (joined_gdf.loc[joined_gdf[apn].isin(single_ids)].set_index(apn)[zone])

    # Duplicates
    # resolve via overlay + largest polygon overlap
    dup_ids = zones_per_apn[zones_per_apn > 1].index
    multi_map = pd.Series(dtype=object)  # allows us to concatenate later if no duplicates

    if len(dup_ids) > 0:
        # Only overlay parcels that had multiple zones in joined_gdf
        dup_parcels = parcels.loc[parcels[apn].isin(dup_ids), [apn, "geometry"]]

        overlaps = gpd.overlay(
            dup_parcels,
            zones[[zone, "geometry"]],
            how="intersection",
            keep_geom_type=False
        )
       
        # projection
        area_crs = parcels.estimate_utm_crs()
        if area_crs is None:
            raise ValueError("Could not estimate a projected CRS for area calculations.")
        ovr_area = overlaps.to_crs(area_crs)

        # calculate areas of intersections, dropping non-polygonal 
        ovr_area["overlap_area"] = ovr_area.geometry.area
        poly_area = ovr_area[ovr_area["overlap_area"] > 0].copy()

        if not poly_area.empty:   
            
            # find largest overlap for duplicate parcels
            idx = poly_area.groupby(apn)["overlap_area"].idxmax()

            # For each duplicate parcel, keep only the zone from the largest overlap and map APN -> zoning
            multi_map = poly_area.loc[idx].set_index(apn)[zone] 


    # Combine single + multi maps; merge back to parcels, ensuring all rows are kept
    final_map = (pd.concat([single_map, multi_map])
                 .groupby(level=0)
                 .last()  # if an APN appears in both, multi_map overrides
                 .rename(zone_out))

    parcels_final = parcels.merge(final_map, left_on=apn, right_index=True, how="left")

    return parcels_final
