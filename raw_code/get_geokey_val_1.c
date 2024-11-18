static char *get_geokey_val(int key, int val)

{

    char *ap;



    if (val == TIFF_GEO_KEY_UNDEFINED)

        return av_strdup("undefined");

    if (val == TIFF_GEO_KEY_USER_DEFINED)

        return av_strdup("User-Defined");



#define RET_GEOKEY_VAL(TYPE, array)\

    if (val >= TIFF_##TYPE##_OFFSET &&\

        val - TIFF_##TYPE##_OFFSET < FF_ARRAY_ELEMS(ff_tiff_##array##_codes))\

        return av_strdup(ff_tiff_##array##_codes[val - TIFF_##TYPE##_OFFSET]);



    switch (key) {

    case TIFF_GT_MODEL_TYPE_GEOKEY:

        RET_GEOKEY_VAL(GT_MODEL_TYPE, gt_model_type);

        break;

    case TIFF_GT_RASTER_TYPE_GEOKEY:

        RET_GEOKEY_VAL(GT_RASTER_TYPE, gt_raster_type);

        break;

    case TIFF_GEOG_LINEAR_UNITS_GEOKEY:

    case TIFF_PROJ_LINEAR_UNITS_GEOKEY:

    case TIFF_VERTICAL_UNITS_GEOKEY:

        RET_GEOKEY_VAL(LINEAR_UNIT, linear_unit);

        break;

    case TIFF_GEOG_ANGULAR_UNITS_GEOKEY:

    case TIFF_GEOG_AZIMUTH_UNITS_GEOKEY:

        RET_GEOKEY_VAL(ANGULAR_UNIT, angular_unit);

        break;

    case TIFF_GEOGRAPHIC_TYPE_GEOKEY:

        RET_GEOKEY_VAL(GCS_TYPE, gcs_type);

        RET_GEOKEY_VAL(GCSE_TYPE, gcse_type);

        break;

    case TIFF_GEOG_GEODETIC_DATUM_GEOKEY:

        RET_GEOKEY_VAL(GEODETIC_DATUM, geodetic_datum);

        RET_GEOKEY_VAL(GEODETIC_DATUM_E, geodetic_datum_e);

        break;

    case TIFF_GEOG_ELLIPSOID_GEOKEY:

        RET_GEOKEY_VAL(ELLIPSOID, ellipsoid);

        break;

    case TIFF_GEOG_PRIME_MERIDIAN_GEOKEY:

        RET_GEOKEY_VAL(PRIME_MERIDIAN, prime_meridian);

        break;

    case TIFF_PROJECTED_CS_TYPE_GEOKEY:

        return av_strdup(search_keyval(ff_tiff_proj_cs_type_codes, FF_ARRAY_ELEMS(ff_tiff_proj_cs_type_codes), val));

        break;

    case TIFF_PROJECTION_GEOKEY:

        return av_strdup(search_keyval(ff_tiff_projection_codes, FF_ARRAY_ELEMS(ff_tiff_projection_codes), val));

        break;

    case TIFF_PROJ_COORD_TRANS_GEOKEY:

        RET_GEOKEY_VAL(COORD_TRANS, coord_trans);

        break;

    case TIFF_VERTICAL_CS_TYPE_GEOKEY:

        RET_GEOKEY_VAL(VERT_CS, vert_cs);

        RET_GEOKEY_VAL(ORTHO_VERT_CS, ortho_vert_cs);

        break;



    }



    ap = av_malloc(14);

    if (ap)

        snprintf(ap, 14, "Unknown-%d", val);

    return ap;

}
