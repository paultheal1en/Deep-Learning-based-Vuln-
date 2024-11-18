static int cp15_tls_load_store(CPUState *env, DisasContext *s, uint32_t insn, uint32_t rd)

{

    TCGv tmp;

    int cpn = (insn >> 16) & 0xf;

    int cpm = insn & 0xf;

    int op = ((insn >> 5) & 7) | ((insn >> 18) & 0x38);



    if (!arm_feature(env, ARM_FEATURE_V6K))

        return 0;



    if (!(cpn == 13 && cpm == 0))

        return 0;



    if (insn & ARM_CP_RW_BIT) {

        switch (op) {

        case 2:

            tmp = load_cpu_field(cp15.c13_tls1);

            break;

        case 3:

            tmp = load_cpu_field(cp15.c13_tls2);

            break;

        case 4:

            tmp = load_cpu_field(cp15.c13_tls3);

            break;

        default:

            return 0;

        }

        store_reg(s, rd, tmp);



    } else {

        tmp = load_reg(s, rd);

        switch (op) {

        case 2:

            store_cpu_field(tmp, cp15.c13_tls1);

            break;

        case 3:

            store_cpu_field(tmp, cp15.c13_tls2);

            break;

        case 4:

            store_cpu_field(tmp, cp15.c13_tls3);

            break;

        default:

            dead_tmp(tmp);

            return 0;

        }

    }

    return 1;

}
