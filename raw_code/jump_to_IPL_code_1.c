static void jump_to_IPL_code(uint64_t address)

{

    /* store the subsystem information _after_ the bootmap was loaded */

    write_subsystem_identification();

    /*

     * The IPL PSW is at address 0. We also must not overwrite the

     * content of non-BIOS memory after we loaded the guest, so we

     * save the original content and restore it in jump_to_IPL_2.

     */

    ResetInfo *current = 0;



    save = *current;

    current->ipl_addr = (uint32_t) (uint64_t) &jump_to_IPL_2;

    current->ipl_continue = address & 0x7fffffff;



    debug_print_int("set IPL addr to", current->ipl_continue);



    /* Ensure the guest output starts fresh */

    sclp_print("\n");



    /*

     * HACK ALERT.

     * We use the load normal reset to keep r15 unchanged. jump_to_IPL_2

     * can then use r15 as its stack pointer.

     */

    asm volatile("lghi 1,1\n\t"

                 "diag 1,1,0x308\n\t"

                 : : : "1", "memory");

    virtio_panic("\n! IPL returns !\n");

}
