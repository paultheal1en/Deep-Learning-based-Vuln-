static inline void gen_op_fcmps(int fccno, TCGv r_rs1, TCGv r_rs2)

{

    gen_helper_fcmps(cpu_env, r_rs1, r_rs2);

}
