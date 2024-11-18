static void gdb_set_cpu_pc(GDBState *s, target_ulong pc)

{

#if defined(TARGET_I386)

    cpu_synchronize_state(s->c_cpu);

    s->c_cpu->eip = pc;

#elif defined (TARGET_PPC)

    s->c_cpu->nip = pc;

#elif defined (TARGET_SPARC)

    s->c_cpu->pc = pc;

    s->c_cpu->npc = pc + 4;

#elif defined (TARGET_ARM)

    s->c_cpu->regs[15] = pc;

#elif defined (TARGET_SH4)

    s->c_cpu->pc = pc;

#elif defined (TARGET_MIPS)

    s->c_cpu->active_tc.PC = pc;

#elif defined (TARGET_MICROBLAZE)

    s->c_cpu->sregs[SR_PC] = pc;

#elif defined (TARGET_CRIS)

    s->c_cpu->pc = pc;

#elif defined (TARGET_ALPHA)

    s->c_cpu->pc = pc;

#elif defined (TARGET_S390X)

    cpu_synchronize_state(s->c_cpu);

    s->c_cpu->psw.addr = pc;

#endif

}
