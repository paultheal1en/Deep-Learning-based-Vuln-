static int of_dpa_cmd_flow_add(OfDpa *of_dpa, uint64_t cookie,

                               RockerTlv **flow_tlvs)

{

    OfDpaFlow *flow = of_dpa_flow_find(of_dpa, cookie);

    int err = ROCKER_OK;



    if (flow) {

        return -ROCKER_EEXIST;

    }



    flow = of_dpa_flow_alloc(cookie);

    if (!flow) {

        return -ROCKER_ENOMEM;

    }



    err = of_dpa_cmd_flow_add_mod(of_dpa, flow, flow_tlvs);

    if (err) {

        g_free(flow);

        return err;

    }



    return of_dpa_flow_add(of_dpa, flow);

}
