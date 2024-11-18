static void rtas_set_indicator(PowerPCCPU *cpu, sPAPRMachineState *spapr,

                               uint32_t token, uint32_t nargs,

                               target_ulong args, uint32_t nret,

                               target_ulong rets)

{

    uint32_t sensor_type;

    uint32_t sensor_index;

    uint32_t sensor_state;

    uint32_t ret = RTAS_OUT_SUCCESS;

    sPAPRDRConnector *drc;

    sPAPRDRConnectorClass *drck;



    if (nargs != 3 || nret != 1) {

        ret = RTAS_OUT_PARAM_ERROR;

        goto out;

    }



    sensor_type = rtas_ld(args, 0);

    sensor_index = rtas_ld(args, 1);

    sensor_state = rtas_ld(args, 2);



    if (!sensor_type_is_dr(sensor_type)) {

        goto out_unimplemented;

    }



    /* if this is a DR sensor we can assume sensor_index == drc_index */

    drc = spapr_drc_by_index(sensor_index);

    if (!drc) {

        trace_spapr_rtas_set_indicator_invalid(sensor_index);

        ret = RTAS_OUT_PARAM_ERROR;

        goto out;

    }

    drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);



    switch (sensor_type) {

    case RTAS_SENSOR_TYPE_ISOLATION_STATE:

        ret = drck->set_isolation_state(drc, sensor_state);

        break;

    case RTAS_SENSOR_TYPE_DR:

        ret = drck->set_indicator_state(drc, sensor_state);

        break;

    case RTAS_SENSOR_TYPE_ALLOCATION_STATE:

        ret = drck->set_allocation_state(drc, sensor_state);

        break;

    default:

        goto out_unimplemented;

    }



out:

    rtas_st(rets, 0, ret);

    return;



out_unimplemented:

    /* currently only DR-related sensors are implemented */

    trace_spapr_rtas_set_indicator_not_supported(sensor_index, sensor_type);

    rtas_st(rets, 0, RTAS_OUT_NOT_SUPPORTED);

}
