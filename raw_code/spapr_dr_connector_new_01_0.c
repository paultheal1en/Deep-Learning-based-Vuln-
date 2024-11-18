sPAPRDRConnector *spapr_dr_connector_new(Object *owner,

                                         sPAPRDRConnectorType type,

                                         uint32_t id)

{

    sPAPRDRConnector *drc =

        SPAPR_DR_CONNECTOR(object_new(TYPE_SPAPR_DR_CONNECTOR));



    g_assert(type);



    drc->type = type;

    drc->id = id;

    drc->owner = owner;

    object_property_add_child(owner, "dr-connector[*]", OBJECT(drc), NULL);

    object_property_set_bool(OBJECT(drc), true, "realized", NULL);



    /* human-readable name for a DRC to encode into the DT

     * description. this is mainly only used within a guest in place

     * of the unique DRC index.

     *

     * in the case of VIO/PCI devices, it corresponds to a

     * "location code" that maps a logical device/function (DRC index)

     * to a physical (or virtual in the case of VIO) location in the

     * system by chaining together the "location label" for each

     * encapsulating component.

     *

     * since this is more to do with diagnosing physical hardware

     * issues than guest compatibility, we choose location codes/DRC

     * names that adhere to the documented format, but avoid encoding

     * the entire topology information into the label/code, instead

     * just using the location codes based on the labels for the

     * endpoints (VIO/PCI adaptor connectors), which is basically

     * just "C" followed by an integer ID.

     *

     * DRC names as documented by PAPR+ v2.7, 13.5.2.4

     * location codes as documented by PAPR+ v2.7, 12.3.1.5

     */

    switch (drc->type) {

    case SPAPR_DR_CONNECTOR_TYPE_CPU:

        drc->name = g_strdup_printf("CPU %d", id);

        break;

    case SPAPR_DR_CONNECTOR_TYPE_PHB:

        drc->name = g_strdup_printf("PHB %d", id);

        break;

    case SPAPR_DR_CONNECTOR_TYPE_VIO:

    case SPAPR_DR_CONNECTOR_TYPE_PCI:

        drc->name = g_strdup_printf("C%d", id);

        break;

    case SPAPR_DR_CONNECTOR_TYPE_LMB:

        drc->name = g_strdup_printf("LMB %d", id);

        break;

    default:

        g_assert(false);

    }



    /* PCI slot always start in a USABLE state, and stay there */

    if (drc->type == SPAPR_DR_CONNECTOR_TYPE_PCI) {

        drc->allocation_state = SPAPR_DR_ALLOCATION_STATE_USABLE;

    }



    return drc;

}
