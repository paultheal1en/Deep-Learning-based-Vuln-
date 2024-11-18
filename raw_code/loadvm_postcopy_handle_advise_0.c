static int loadvm_postcopy_handle_advise(MigrationIncomingState *mis)

{

    PostcopyState ps = postcopy_state_set(POSTCOPY_INCOMING_ADVISE);

    uint64_t remote_pagesize_summary, local_pagesize_summary, remote_tps;



    trace_loadvm_postcopy_handle_advise();

    if (ps != POSTCOPY_INCOMING_NONE) {

        error_report("CMD_POSTCOPY_ADVISE in wrong postcopy state (%d)", ps);

        return -1;

    }



    if (!migrate_postcopy_ram()) {

        return 0;

    }



    if (!postcopy_ram_supported_by_host()) {

        postcopy_state_set(POSTCOPY_INCOMING_NONE);

        return -1;

    }



    remote_pagesize_summary = qemu_get_be64(mis->from_src_file);

    local_pagesize_summary = ram_pagesize_summary();



    if (remote_pagesize_summary != local_pagesize_summary)  {

        /*

         * This detects two potential causes of mismatch:

         *   a) A mismatch in host page sizes

         *      Some combinations of mismatch are probably possible but it gets

         *      a bit more complicated.  In particular we need to place whole

         *      host pages on the dest at once, and we need to ensure that we

         *      handle dirtying to make sure we never end up sending part of

         *      a hostpage on it's own.

         *   b) The use of different huge page sizes on source/destination

         *      a more fine grain test is performed during RAM block migration

         *      but this test here causes a nice early clear failure, and

         *      also fails when passed to an older qemu that doesn't

         *      do huge pages.

         */

        error_report("Postcopy needs matching RAM page sizes (s=%" PRIx64

                                                             " d=%" PRIx64 ")",

                     remote_pagesize_summary, local_pagesize_summary);

        return -1;

    }



    remote_tps = qemu_get_be64(mis->from_src_file);

    if (remote_tps != qemu_target_page_size()) {

        /*

         * Again, some differences could be dealt with, but for now keep it

         * simple.

         */

        error_report("Postcopy needs matching target page sizes (s=%d d=%zd)",

                     (int)remote_tps, qemu_target_page_size());

        return -1;

    }



    if (ram_postcopy_incoming_init(mis)) {

        return -1;

    }



    postcopy_state_set(POSTCOPY_INCOMING_ADVISE);



    return 0;

}
