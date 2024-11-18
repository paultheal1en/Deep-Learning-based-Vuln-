void do_unassigned_access(target_phys_addr_t addr, int is_write, int is_exec,

                          int unused, int size)

{

    if (is_exec)

        helper_raise_exception(EXCP_IBE);

    else

        helper_raise_exception(EXCP_DBE);

}
