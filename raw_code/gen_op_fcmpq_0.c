static inline void gen_op_fcmpq(int fccno)

{

    switch (fccno) {

    case 0:

        gen_helper_fcmpq(cpu_env);

        break;

    case 1:

        gen_helper_fcmpq_fcc1(cpu_env);

        break;

    case 2:

        gen_helper_fcmpq_fcc2(cpu_env);

        break;

    case 3:

        gen_helper_fcmpq_fcc3(cpu_env);

        break;

    }

}
