static ExitStatus gen_mfpr(TCGv va, int regno)

{

    int data = cpu_pr_data(regno);



    /* Special help for VMTIME and WALLTIME.  */

    if (regno == 250 || regno == 249) {

	void (*helper)(TCGv) = gen_helper_get_walltime;

	if (regno == 249) {

		helper = gen_helper_get_vmtime;

	}

        if (use_icount) {

            gen_io_start();

            helper(va);

            gen_io_end();

            return EXIT_PC_STALE;

        } else {

            helper(va);

            return NO_EXIT;

        }

    }



    /* The basic registers are data only, and unknown registers

       are read-zero, write-ignore.  */

    if (data == 0) {

        tcg_gen_movi_i64(va, 0);

    } else if (data & PR_BYTE) {

        tcg_gen_ld8u_i64(va, cpu_env, data & ~PR_BYTE);

    } else if (data & PR_LONG) {

        tcg_gen_ld32s_i64(va, cpu_env, data & ~PR_LONG);

    } else {

        tcg_gen_ld_i64(va, cpu_env, data);

    }

    return NO_EXIT;

}
