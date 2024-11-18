static inline void gen_op_fcmpes(int fccno, TCGv r_rs1, TCGv r_rs2)

{

    gen_helper_fcmpes(cpu_env, r_rs1, r_rs2);

}
