static uint64_t get_migration_pass(void)

{

    QDict *rsp, *rsp_return, *rsp_ram;

    uint64_t result;



    rsp = return_or_event(qmp("{ 'execute': 'query-migrate' }"));

    rsp_return = qdict_get_qdict(rsp, "return");

    if (!qdict_haskey(rsp_return, "ram")) {

        /* Still in setup */

        result = 0;

    } else {

        rsp_ram = qdict_get_qdict(rsp_return, "ram");

        result = qdict_get_try_int(rsp_ram, "dirty-sync-count", 0);

        QDECREF(rsp);

    }

    return result;

}
