static void init_proc_750 (CPUPPCState *env)

{

    gen_spr_ne_601(env);

    gen_spr_7xx(env);

    /* XXX : not implemented */

    spr_register(env, SPR_L2CR, "L2CR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, NULL,

                 0x00000000);

    /* Time base */

    gen_tbl(env);

    /* Thermal management */

    gen_spr_thrm(env);

    /* Hardware implementation registers */

    /* XXX : not implemented */

    spr_register(env, SPR_HID0, "HID0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_HID1, "HID1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* Memory management */

    gen_low_BATs(env);

    /* XXX: high BATs are also present but are known to be bugged on

     *      die version 1.x

     */

    init_excp_7x0(env);

    env->dcache_line_size = 32;

    env->icache_line_size = 32;

    /* Allocate hardware IRQ controller */

    ppc6xx_irq_init(env);

}
