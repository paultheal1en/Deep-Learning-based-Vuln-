static void ppc405_ocm_init(CPUPPCState *env)

{

    ppc405_ocm_t *ocm;



    ocm = g_malloc0(sizeof(ppc405_ocm_t));

    /* XXX: Size is 4096 or 0x04000000 */

    memory_region_init_ram(&ocm->isarc_ram, NULL, "ppc405.ocm", 4096,

                           &error_abort);

    vmstate_register_ram_global(&ocm->isarc_ram);

    memory_region_init_alias(&ocm->dsarc_ram, NULL, "ppc405.dsarc", &ocm->isarc_ram,

                             0, 4096);

    qemu_register_reset(&ocm_reset, ocm);

    ppc_dcr_register(env, OCM0_ISARC,

                     ocm, &dcr_read_ocm, &dcr_write_ocm);

    ppc_dcr_register(env, OCM0_ISACNTL,

                     ocm, &dcr_read_ocm, &dcr_write_ocm);

    ppc_dcr_register(env, OCM0_DSARC,

                     ocm, &dcr_read_ocm, &dcr_write_ocm);

    ppc_dcr_register(env, OCM0_DSACNTL,

                     ocm, &dcr_read_ocm, &dcr_write_ocm);

}
