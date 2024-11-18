static int disas_cp14_write(CPUState * env, DisasContext *s, uint32_t insn)

{

    int crn = (insn >> 16) & 0xf;

    int crm = insn & 0xf;

    int op1 = (insn >> 21) & 7;

    int op2 = (insn >> 5) & 7;

    int rt = (insn >> 12) & 0xf;

    TCGv tmp;



    if (arm_feature(env, ARM_FEATURE_THUMB2EE)) {

        if (op1 == 6 && crn == 0 && crm == 0 && op2 == 0) {

            /* TEECR */

            if (IS_USER(s))

                return 1;

            tmp = load_reg(s, rt);

            gen_helper_set_teecr(cpu_env, tmp);

            dead_tmp(tmp);

            return 0;

        }

        if (op1 == 6 && crn == 1 && crm == 0 && op2 == 0) {

            /* TEEHBR */

            if (IS_USER(s) && (env->teecr & 1))

                return 1;

            tmp = load_reg(s, rt);

            store_cpu_field(tmp, teehbr);

            return 0;

        }

    }

    fprintf(stderr, "Unknown cp14 write op1:%d crn:%d crm:%d op2:%d\n",

            op1, crn, crm, op2);

    return 1;

}
