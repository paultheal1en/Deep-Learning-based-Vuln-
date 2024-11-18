static int of_dpa_cmd_group_add(OfDpa *of_dpa, uint32_t group_id,

                                RockerTlv **group_tlvs)

{

    OfDpaGroup *group = of_dpa_group_find(of_dpa, group_id);

    int err;



    if (group) {

        return -ROCKER_EEXIST;

    }



    group = of_dpa_group_alloc(group_id);

    if (!group) {

        return -ROCKER_ENOMEM;

    }



    err = of_dpa_cmd_group_do(of_dpa, group_id, group, group_tlvs);

    if (err) {

        goto err_cmd_add;

    }



    err = of_dpa_group_add(of_dpa, group);

    if (err) {

        goto err_cmd_add;

    }



    return ROCKER_OK;



err_cmd_add:

    g_free(group);

    return err;

}
