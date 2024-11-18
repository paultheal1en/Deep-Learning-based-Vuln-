void tb_invalidate_page_range(target_ulong start, target_ulong end)

{

    /* XXX: cannot enable it yet because it yields to MMU exception

       where NIP != read address on PowerPC */

#if 0

    target_ulong phys_addr;

    phys_addr = get_phys_addr_code(env, start);

    tb_invalidate_phys_page_range(phys_addr, phys_addr + end - start, 0);

#endif

}
