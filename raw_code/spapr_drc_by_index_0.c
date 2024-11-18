sPAPRDRConnector *spapr_drc_by_index(uint32_t index)

{

    Object *obj;

    char name[256];



    snprintf(name, sizeof(name), "%s/%x", DRC_CONTAINER_PATH, index);

    obj = object_resolve_path(name, NULL);



    return !obj ? NULL : SPAPR_DR_CONNECTOR(obj);

}
