static void init_ppc_proc (CPUPPCState *env, ppc_def_t *def)

{

    env->reserve = -1;

    /* Default MMU definitions */

    env->nb_BATs = -1;

    env->nb_tlb = 0;

    env->nb_ways = 0;

    /* XXX: missing:

     * 32 bits PowerPC:

     * - MPC5xx(x)

     * - MPC8xx(x)

     * - RCPU (same as MPC5xx ?)

     */

    spr_register(env, SPR_PVR, "PVR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, SPR_NOACCESS,

                 def->pvr);

    printf("%s: PVR %08x mask %08x => %08x\n", __func__,

           def->pvr, def->pvr_mask, def->pvr & def->pvr_mask);

    switch (def->pvr) {

    /* Embedded PowerPC from IBM                           */

    case CPU_PPC_401A1:   /* 401 A1 family                 */

    case CPU_PPC_401B2:   /* 401 B2 family                 */

    case CPU_PPC_401C2:   /* 401 C2 family                 */

    case CPU_PPC_401D2:   /* 401 D2 family                 */

    case CPU_PPC_401E2:   /* 401 E2 family                 */

    case CPU_PPC_401F2:   /* 401 F2 family                 */

    case CPU_PPC_401G2:   /* 401 G2 family                 */

    case CPU_PPC_IOP480:  /* IOP 480 family                */

    case CPU_PPC_COBRA:   /* IBM Processor for Network Resources */

        gen_spr_generic(env);

        gen_spr_40x(env);

        gen_spr_401_403(env);

#if defined (TODO)

        /* XXX: optional ? */

        gen_spr_compress(env);

#endif

        env->nb_BATs = 0;

        env->nb_tlb = 64;

        env->nb_ways = 1;

        env->id_tlbs = 0;

        /* XXX: TODO: allocate internal IRQ controller */

        break;



    case CPU_PPC_403GA:   /* 403 GA family                 */

    case CPU_PPC_403GB:   /* 403 GB family                 */

    case CPU_PPC_403GC:   /* 403 GC family                 */

    case CPU_PPC_403GCX:  /* 403 GCX family                */

        gen_spr_generic(env);

        gen_spr_40x(env);

        gen_spr_401_403(env);

        gen_spr_403(env);

        env->nb_BATs = 0;

        env->nb_tlb = 64;

        env->nb_ways = 1;

        env->id_tlbs = 0;

        /* XXX: TODO: allocate internal IRQ controller */

        break;



    case CPU_PPC_405CR:   /* 405 GP/CR family              */

    case CPU_PPC_405EP:   /* 405 EP family                 */

    case CPU_PPC_405GPR:  /* 405 GPR family                */

    case CPU_PPC_405D2:   /* 405 D2 family                 */

    case CPU_PPC_405D4:   /* 405 D4 family                 */

        gen_spr_generic(env);

        /* Time base */

        gen_tbl(env);

        gen_spr_40x(env);

        gen_spr_405(env);

        env->nb_BATs = 0;

        env->nb_tlb = 64;

        env->nb_ways = 1;

        env->id_tlbs = 0;

        /* Allocate hardware IRQ controller */

        ppc405_irq_init(env);

        break;



    case CPU_PPC_NPE405H: /* NPe405 H family               */

    case CPU_PPC_NPE405H2:

    case CPU_PPC_NPE405L: /* Npe405 L family               */

        gen_spr_generic(env);

        /* Time base */

        gen_tbl(env);

        gen_spr_40x(env);

        gen_spr_405(env);

        env->nb_BATs = 0;

        env->nb_tlb = 64;

        env->nb_ways = 1;

        env->id_tlbs = 0;

        /* Allocate hardware IRQ controller */

        ppc405_irq_init(env);

        break;



#if defined (TODO)

    case CPU_PPC_STB01000:

#endif

#if defined (TODO)

    case CPU_PPC_STB01010:

#endif

#if defined (TODO)

    case CPU_PPC_STB0210:

#endif

    case CPU_PPC_STB03:   /* STB03 family                  */

#if defined (TODO)

    case CPU_PPC_STB043:  /* STB043 family                  */

#endif

#if defined (TODO)

    case CPU_PPC_STB045:  /* STB045 family                  */

#endif

    case CPU_PPC_STB25:   /* STB25 family                  */

#if defined (TODO)

    case CPU_PPC_STB130:  /* STB130 family                 */

#endif

        gen_spr_generic(env);

        /* Time base */

        gen_tbl(env);

        gen_spr_40x(env);

        gen_spr_405(env);

        env->nb_BATs = 0;

        env->nb_tlb = 64;

        env->nb_ways = 1;

        env->id_tlbs = 0;

        /* Allocate hardware IRQ controller */

        ppc405_irq_init(env);

        break;



    case CPU_PPC_440EP:   /* 440 EP family                 */

    case CPU_PPC_440GP:   /* 440 GP family                 */

    case CPU_PPC_440GX:   /* 440 GX family                 */

    case CPU_PPC_440GXc:  /* 440 GXc family                */

    case CPU_PPC_440GXf:  /* 440 GXf family                */

    case CPU_PPC_440SP:   /* 440 SP family                 */

    case CPU_PPC_440SP2:

    case CPU_PPC_440SPE:  /* 440 SPE family                */

        gen_spr_generic(env);

        /* Time base */

        gen_tbl(env);

        gen_spr_BookE(env);

        gen_spr_440(env);

        env->nb_BATs = 0;

        env->nb_tlb = 64;

        env->nb_ways = 1;

        env->id_tlbs = 0;

        /* XXX: TODO: allocate internal IRQ controller */

        break;



    /* Embedded PowerPC from Freescale                     */

#if defined (TODO)

    case CPU_PPC_5xx:

        break;

#endif

#if defined (TODO)

    case CPU_PPC_8xx:     /* MPC821 / 823 / 850 / 860      */

        break;

#endif

#if defined (TODO)

    case CPU_PPC_82xx_HIP3:    /* MPC8240 / 8260                */

    case CPU_PPC_82xx_HIP4:    /* MPC8240 / 8260                */

        break;

#endif

#if defined (TODO)

    case CPU_PPC_827x:    /* MPC 827x / 828x               */

        break;

#endif



    /* XXX: Use MPC8540 PVR to implement a test PowerPC BookE target */

    case CPU_PPC_e500v110:

    case CPU_PPC_e500v120:

    case CPU_PPC_e500v210:

    case CPU_PPC_e500v220:

        gen_spr_generic(env);

        /* Time base */

        gen_tbl(env);

        gen_spr_BookE(env);

        gen_spr_BookE_FSL(env);

        env->nb_BATs = 0;

        env->nb_tlb = 64;

        env->nb_ways = 1;

        env->id_tlbs = 0;

        /* XXX: TODO: allocate internal IRQ controller */

        break;



#if defined (TODO)

    case CPU_PPC_e600:

        break;

#endif



    /* 32 bits PowerPC                                     */

    case CPU_PPC_601:     /* PowerPC 601                   */

        gen_spr_generic(env);

        gen_spr_ne_601(env);

        gen_spr_601(env);

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

        /* XXX : not implemented */

        spr_register(env, SPR_601_HID2, "HID2",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     0x00000000);

        /* XXX : not implemented */

        spr_register(env, SPR_601_HID5, "HID5",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     0x00000000);

        /* XXX : not implemented */

#if 0 /* ? */

        spr_register(env, SPR_601_HID15, "HID15",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     0x00000000);

#endif

        env->nb_tlb = 64;

        env->nb_ways = 2;

        env->id_tlbs = 0;

        env->id_tlbs = 0;

        /* XXX: TODO: allocate internal IRQ controller */

        break;



    case CPU_PPC_602:     /* PowerPC 602                   */

        gen_spr_generic(env);

        gen_spr_ne_601(env);

        /* Memory management */

        gen_low_BATs(env);

        /* Time base */

        gen_tbl(env);

        gen_6xx_7xx_soft_tlb(env, 64, 2);

        gen_spr_602(env);

        /* hardware implementation registers */

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

        /* Allocate hardware IRQ controller */

        ppc6xx_irq_init(env);

        break;



    case CPU_PPC_603:     /* PowerPC 603                   */

    case CPU_PPC_603E:    /* PowerPC 603e                  */

    case CPU_PPC_603E7v:

    case CPU_PPC_603E7v2:

    case CPU_PPC_603P:    /* PowerPC 603p                  */

    case CPU_PPC_603R:    /* PowerPC 603r                  */

        gen_spr_generic(env);

        gen_spr_ne_601(env);

        /* Memory management */

        gen_low_BATs(env);

        /* Time base */

        gen_tbl(env);

        gen_6xx_7xx_soft_tlb(env, 64, 2);

        gen_spr_603(env);

        /* hardware implementation registers */

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

        /* Allocate hardware IRQ controller */

        ppc6xx_irq_init(env);

        break;



    case CPU_PPC_G2:      /* PowerPC G2 family             */

    case CPU_PPC_G2H4:

    case CPU_PPC_G2gp:

    case CPU_PPC_G2ls:

    case CPU_PPC_G2LE:    /* PowerPC G2LE family           */

    case CPU_PPC_G2LEgp:

    case CPU_PPC_G2LEls:

        gen_spr_generic(env);

        gen_spr_ne_601(env);

        /* Memory management */

        gen_low_BATs(env);

        /* Time base */

        gen_tbl(env);

        /* Memory management */

        gen_high_BATs(env);

        gen_6xx_7xx_soft_tlb(env, 64, 2);

        gen_spr_G2_755(env);

        gen_spr_G2(env);

        /* Hardware implementation register */

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

        /* XXX : not implemented */

        spr_register(env, SPR_HID2, "HID2",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     0x00000000);

        /* Allocate hardware IRQ controller */

        ppc6xx_irq_init(env);

        break;



    case CPU_PPC_604:     /* PowerPC 604                   */

    case CPU_PPC_604E:    /* PowerPC 604e                  */

    case CPU_PPC_604R:    /* PowerPC 604r                  */

        gen_spr_generic(env);

        gen_spr_ne_601(env);

        /* Memory management */

        gen_low_BATs(env);

        /* Time base */

        gen_tbl(env);

        gen_spr_604(env);

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

        /* Allocate hardware IRQ controller */

        ppc6xx_irq_init(env);

        break;



    case CPU_PPC_74x:     /* PowerPC 740 / 750             */

    case CPU_PPC_740E:

    case CPU_PPC_750E:

    case CPU_PPC_74xP:    /* PowerPC 740P / 750P           */

    case CPU_PPC_750CXE21: /* IBM PowerPC 750cxe            */

    case CPU_PPC_750CXE22:

    case CPU_PPC_750CXE23:

    case CPU_PPC_750CXE24:

    case CPU_PPC_750CXE24b:

    case CPU_PPC_750CXE31:

    case CPU_PPC_750CXE31b:

    case CPU_PPC_750CXR:

        gen_spr_generic(env);

        gen_spr_ne_601(env);

        /* Memory management */

        gen_low_BATs(env);

        /* Time base */

        gen_tbl(env);

        gen_spr_7xx(env);

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

        /* Allocate hardware IRQ controller */

        ppc6xx_irq_init(env);

        break;



    case CPU_PPC_750FX10: /* IBM PowerPC 750 FX            */

    case CPU_PPC_750FX20:

    case CPU_PPC_750FX21:

    case CPU_PPC_750FX22:

    case CPU_PPC_750FX23:

    case CPU_PPC_750GX10: /* IBM PowerPC 750 GX            */

    case CPU_PPC_750GX11:

    case CPU_PPC_750GX12:

        gen_spr_generic(env);

        gen_spr_ne_601(env);

        /* Memory management */

        gen_low_BATs(env);

        /* PowerPC 750fx & 750gx has 8 DBATs and 8 IBATs */

        gen_high_BATs(env);

        /* Time base */

        gen_tbl(env);

        gen_spr_7xx(env);

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

        /* XXX : not implemented */

        spr_register(env, SPR_750_HID2, "HID2",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     0x00000000);

        /* Allocate hardware IRQ controller */

        ppc6xx_irq_init(env);

        break;



    case CPU_PPC_755_10:  /* PowerPC 755                   */

    case CPU_PPC_755_11:

    case CPU_PPC_755_20:

    case CPU_PPC_755D:

    case CPU_PPC_755E:

        gen_spr_generic(env);

        gen_spr_ne_601(env);

        /* Memory management */

        gen_low_BATs(env);

        /* Time base */

        gen_tbl(env);

        /* Memory management */

        gen_high_BATs(env);

        gen_6xx_7xx_soft_tlb(env, 64, 2);

        gen_spr_G2_755(env);

        /* L2 cache control */

        /* XXX : not implemented */

        spr_register(env, SPR_ICTC, "ICTC",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     0x00000000);

        /* XXX : not implemented */

        spr_register(env, SPR_L2PM, "L2PM",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     0x00000000);

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

        /* XXX : not implemented */

        spr_register(env, SPR_HID2, "HID2",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     0x00000000);

        /* Allocate hardware IRQ controller */

        ppc6xx_irq_init(env);

        break;



#if defined (TODO)

    /* G4 family */

    case CPU_PPC_7400:    /* PowerPC 7400                  */

    case CPU_PPC_7410C:   /* PowerPC 7410                  */

    case CPU_PPC_7410D:

    case CPU_PPC_7410E:

    case CPU_PPC_7441:    /* PowerPC 7441                  */

    case CPU_PPC_7445:    /* PowerPC 7445                  */

    case CPU_PPC_7447:    /* PowerPC 7447                  */

    case CPU_PPC_7447A:   /* PowerPC 7447A                 */

    case CPU_PPC_7448:    /* PowerPC 7448                  */

    case CPU_PPC_7450:    /* PowerPC 7450                  */

    case CPU_PPC_7450b:

    case CPU_PPC_7451:    /* PowerPC 7451                  */

    case CPU_PPC_7451G:

    case CPU_PPC_7455:    /* PowerPC 7455                  */

    case CPU_PPC_7455F:

    case CPU_PPC_7455G:

    case CPU_PPC_7457:    /* PowerPC 7457                  */

    case CPU_PPC_7457C:

    case CPU_PPC_7457A:   /* PowerPC 7457A                 */

        break;

#endif



    /* 64 bits PowerPC                                     */

#if defined (TARGET_PPC64)

#if defined (TODO)

    case CPU_PPC_620:     /* PowerPC 620                   */

    case CPU_PPC_630:     /* PowerPC 630 (Power 3)         */

    case CPU_PPC_631:     /* PowerPC 631 (Power 3+)        */

    case CPU_PPC_POWER4:  /* Power 4                       */

    case CPU_PPC_POWER4P: /* Power 4+                      */

    case CPU_PPC_POWER5:  /* Power 5                       */

    case CPU_PPC_POWER5P: /* Power 5+                      */

#endif

        break;



    case CPU_PPC_970:     /* PowerPC 970                   */

    case CPU_PPC_970FX10: /* PowerPC 970 FX                */

    case CPU_PPC_970FX20:

    case CPU_PPC_970FX21:

    case CPU_PPC_970FX30:

    case CPU_PPC_970FX31:

    case CPU_PPC_970MP10: /* PowerPC 970 MP                */

    case CPU_PPC_970MP11:

        gen_spr_generic(env);

        gen_spr_ne_601(env);

        /* XXX: not correct */

        gen_low_BATs(env);

        /* Time base */

        gen_tbl(env);

        gen_spr_7xx(env);

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

        /* XXX : not implemented */

        spr_register(env, SPR_750_HID2, "HID2",

                     SPR_NOACCESS, SPR_NOACCESS,

                     &spr_read_generic, &spr_write_generic,

                     0x00000000);

        /* Allocate hardware IRQ controller */

        ppc970_irq_init(env);

        break;



#if defined (TODO)

    case CPU_PPC_CELL10:  /* Cell family                   */

    case CPU_PPC_CELL20:

    case CPU_PPC_CELL30:

    case CPU_PPC_CELL31:

#endif

        break;



#if defined (TODO)

    case CPU_PPC_RS64:    /* Apache (RS64/A35)             */

    case CPU_PPC_RS64II:  /* NorthStar (RS64-II/A50)       */

    case CPU_PPC_RS64III: /* Pulsar (RS64-III)             */

    case CPU_PPC_RS64IV:  /* IceStar/IStar/SStar (RS64-IV) */

#endif

        break;

#endif /* defined (TARGET_PPC64) */



#if defined (TODO)

        /* POWER                                               */

    case CPU_POWER:       /* POWER                         */

    case CPU_POWER2:      /* POWER2                        */

        break;

#endif



    default:

        gen_spr_generic(env);

        /* XXX: TODO: allocate internal IRQ controller */

        break;

    }

    if (env->nb_BATs == -1)

        env->nb_BATs = 4;

    /* Allocate TLBs buffer when needed */

    if (env->nb_tlb != 0) {

        int nb_tlb = env->nb_tlb;

        if (env->id_tlbs != 0)

            nb_tlb *= 2;

        env->tlb = qemu_mallocz(nb_tlb * sizeof(ppc_tlb_t));

        /* Pre-compute some useful values */

        env->tlb_per_way = env->nb_tlb / env->nb_ways;

    }

}
