void HELPER(entry)(CPUXtensaState *env, uint32_t pc, uint32_t s, uint32_t imm)
{
    int callinc = (env->sregs[PS] & PS_CALLINC) >> PS_CALLINC_SHIFT;
    if (s > 3 || ((env->sregs[PS] & (PS_WOE | PS_EXCM)) ^ PS_WOE) != 0) {
        qemu_log("Illegal entry instruction(pc = %08x), PS = %08x\n",
                pc, env->sregs[PS]);
        HELPER(exception_cause)(env, pc, ILLEGAL_INSTRUCTION_CAUSE);
    } else {
        env->regs[(callinc << 2) | (s & 3)] = env->regs[s] - (imm << 3);
        rotate_window(env, callinc);
        env->sregs[WINDOW_START] |=
            windowstart_bit(env->sregs[WINDOW_BASE], env);