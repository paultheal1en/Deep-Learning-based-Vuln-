static void gen_spr_BookE206(CPUPPCState *env, uint32_t mas_mask,

                              uint32_t *tlbncfg)

{

#if !defined(CONFIG_USER_ONLY)

    const char *mas_names[8] = {

        "MAS0", "MAS1", "MAS2", "MAS3", "MAS4", "MAS5", "MAS6", "MAS7",

    };

    int mas_sprn[8] = {

        SPR_BOOKE_MAS0, SPR_BOOKE_MAS1, SPR_BOOKE_MAS2, SPR_BOOKE_MAS3,

        SPR_BOOKE_MAS4, SPR_BOOKE_MAS5, SPR_BOOKE_MAS6, SPR_BOOKE_MAS7,

    };

    int i;



    /* TLB assist registers */

    /* XXX : not implemented */

    for (i = 0; i < 8; i++) {

        if (mas_mask & (1 << i)) {

            spr_register(env, mas_sprn[i], mas_names[i],

                         SPR_NOACCESS, SPR_NOACCESS,

                         &spr_read_generic, &spr_write_generic,

                         0x00000000);

        }

    }

    if (env->nb_pids > 1) {

        /* XXX : not implemented */

        spr_register(env, SPR_BOOKE_PID1, "PID1",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_booke_pid,

                     0x00000000);

    }

    if (env->nb_pids > 2) {

        /* XXX : not implemented */

        spr_register(env, SPR_BOOKE_PID2, "PID2",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_booke_pid,

                     0x00000000);

    }

    /* XXX : not implemented */

    spr_register(env, SPR_MMUCFG, "MMUCFG",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, SPR_NOACCESS,

                 0x00000000); /* TOFIX */

    switch (env->nb_ways) {

    case 4:

        spr_register(env, SPR_BOOKE_TLB3CFG, "TLB3CFG",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, SPR_NOACCESS,

                     tlbncfg[3]);

        /* Fallthru */

    case 3:

        spr_register(env, SPR_BOOKE_TLB2CFG, "TLB2CFG",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, SPR_NOACCESS,

                     tlbncfg[2]);

        /* Fallthru */

    case 2:

        spr_register(env, SPR_BOOKE_TLB1CFG, "TLB1CFG",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, SPR_NOACCESS,

                     tlbncfg[1]);

        /* Fallthru */

    case 1:

        spr_register(env, SPR_BOOKE_TLB0CFG, "TLB0CFG",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, SPR_NOACCESS,

                     tlbncfg[0]);

        /* Fallthru */

    case 0:

    default:

        break;

    }

#endif



    gen_spr_usprgh(env);

}
