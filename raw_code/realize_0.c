static void realize(DeviceState *d, Error **errp)

{

    sPAPRDRConnector *drc = SPAPR_DR_CONNECTOR(d);

    Object *root_container;

    char link_name[256];

    gchar *child_name;

    Error *err = NULL;



    trace_spapr_drc_realize(spapr_drc_index(drc));

    /* NOTE: we do this as part of realize/unrealize due to the fact

     * that the guest will communicate with the DRC via RTAS calls

     * referencing the global DRC index. By unlinking the DRC

     * from DRC_CONTAINER_PATH/<drc_index> we effectively make it

     * inaccessible by the guest, since lookups rely on this path

     * existing in the composition tree

     */

    root_container = container_get(object_get_root(), DRC_CONTAINER_PATH);

    snprintf(link_name, sizeof(link_name), "%x", spapr_drc_index(drc));

    child_name = object_get_canonical_path_component(OBJECT(drc));

    trace_spapr_drc_realize_child(spapr_drc_index(drc), child_name);

    object_property_add_alias(root_container, link_name,

                              drc->owner, child_name, &err);

    g_free(child_name);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    vmstate_register(DEVICE(drc), spapr_drc_index(drc), &vmstate_spapr_drc,

                     drc);

    qemu_register_reset(drc_reset, drc);

    trace_spapr_drc_realize_complete(spapr_drc_index(drc));

}
