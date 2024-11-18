int arm_cpu_write_elf32_note(WriteCoreDumpFunction f, CPUState *cs,

                             int cpuid, void *opaque)

{

    struct arm_note note;

    CPUARMState *env = &ARM_CPU(cs)->env;

    DumpState *s = opaque;

    int ret, i;



    arm_note_init(&note, s, "CORE", 5, NT_PRSTATUS, sizeof(note.prstatus));



    note.prstatus.pr_pid = cpu_to_dump32(s, cpuid);



    for (i = 0; i < 16; ++i) {

        note.prstatus.pr_reg.regs[i] = cpu_to_dump32(s, env->regs[i]);

    }

    note.prstatus.pr_reg.regs[16] = cpu_to_dump32(s, cpsr_read(env));



    ret = f(&note, ARM_PRSTATUS_NOTE_SIZE, s);

    if (ret < 0) {

        return -1;

    }



    return 0;

}
