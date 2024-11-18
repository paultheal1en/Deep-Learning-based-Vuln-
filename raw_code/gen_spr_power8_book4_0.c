static void gen_spr_power8_book4(CPUPPCState *env)

{

    /* Add a number of P8 book4 registers */

#if !defined(CONFIG_USER_ONLY)

    spr_register_kvm(env, SPR_ACOP, "ACOP",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     KVM_REG_PPC_ACOP, 0);

    spr_register_kvm(env, SPR_BOOKS_PID, "PID",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     KVM_REG_PPC_PID, 0);

    spr_register_kvm(env, SPR_WORT, "WORT",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     KVM_REG_PPC_WORT, 0);

#endif

}
