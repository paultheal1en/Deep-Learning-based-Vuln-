int cpu_ppc_handle_mmu_fault (CPUState *env, uint32_t address, int rw,

                              int is_user, int is_softmmu)

{

    uint32_t physical;

    int prot;

    int exception = 0, error_code = 0;

    int access_type;

    int ret = 0;



//    printf("%s 0\n", __func__);

    access_type = env->access_type;

    if (env->user_mode_only) {

        /* user mode only emulation */

        ret = -2;

        goto do_fault;

    }

    /* NASTY BUG workaround */

    if (access_type == ACCESS_CODE && rw) {

	printf("%s: ERROR WRITE CODE ACCESS\n", __func__);

	access_type = ACCESS_INT;

    }

    ret = get_physical_address(env, &physical, &prot,

                               address, rw, access_type);

    if (ret == 0) {

	ret = tlb_set_page(env, address & ~0xFFF, physical, prot,

			   is_user, is_softmmu);

    } else if (ret < 0) {

    do_fault:

#if defined (DEBUG_MMU)

	if (loglevel > 0)

	    cpu_ppc_dump_state(env, logfile, 0);

#endif

        if (access_type == ACCESS_CODE) {

            exception = EXCP_ISI;

            switch (ret) {

            case -1:

                /* No matches in page tables */

                error_code = EXCP_ISI_TRANSLATE;

                break;

            case -2:

                /* Access rights violation */

                error_code = EXCP_ISI_PROT;

                break;

            case -3:

		/* No execute protection violation */

                error_code = EXCP_ISI_NOEXEC;

                break;

            case -4:

                /* Direct store exception */

                /* No code fetch is allowed in direct-store areas */

                error_code = EXCP_ISI_DIRECT;

                break;

            }

        } else {

            exception = EXCP_DSI;

            switch (ret) {

            case -1:

                /* No matches in page tables */

                error_code = EXCP_DSI_TRANSLATE;

                break;

            case -2:

                /* Access rights violation */

                error_code = EXCP_DSI_PROT;

                break;

            case -4:

                /* Direct store exception */

                switch (access_type) {

                case ACCESS_FLOAT:

                    /* Floating point load/store */

                    exception = EXCP_ALIGN;

                    error_code = EXCP_ALIGN_FP;

                    break;

                case ACCESS_RES:

                    /* lwarx, ldarx or srwcx. */

                    exception = EXCP_DSI;

                    error_code = EXCP_DSI_NOTSUP | EXCP_DSI_DIRECT;

                    break;

                case ACCESS_EXT:

                    /* eciwx or ecowx */

                    exception = EXCP_DSI;

                    error_code = EXCP_DSI_NOTSUP | EXCP_DSI_DIRECT |

			EXCP_DSI_ECXW;

                    break;

                default:

		    printf("DSI: invalid exception (%d)\n", ret);

                    exception = EXCP_PROGRAM;

                    error_code = EXCP_INVAL | EXCP_INVAL_INVAL;

                    break;

                }

            }

            if (rw)

                error_code |= EXCP_DSI_STORE;

	    /* Store fault address */

	    env->spr[DAR] = address;

        }

#if 0

        printf("%s: set exception to %d %02x\n",

               __func__, exception, error_code);

#endif

        env->exception_index = exception;

        env->error_code = error_code;

        ret = 1;

    }



    return ret;

}
