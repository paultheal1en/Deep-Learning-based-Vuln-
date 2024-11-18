void monitor_disas(Monitor *mon, CPUState *cpu,

                   target_ulong pc, int nb_insn, int is_physical)

{

    CPUClass *cc = CPU_GET_CLASS(cpu);

    int count, i;

    CPUDebug s;



    INIT_DISASSEMBLE_INFO(s.info, (FILE *)mon, monitor_fprintf);



    s.cpu = cpu;

    monitor_disas_is_physical = is_physical;

    s.info.read_memory_func = monitor_read_memory;

    s.info.print_address_func = generic_print_address;

    s.info.buffer_vma = pc;

    s.info.cap_arch = -1;

    s.info.cap_mode = 0;



#ifdef TARGET_WORDS_BIGENDIAN

    s.info.endian = BFD_ENDIAN_BIG;

#else

    s.info.endian = BFD_ENDIAN_LITTLE;

#endif



    if (cc->disas_set_info) {

        cc->disas_set_info(cpu, &s.info);

    }



    if (s.info.cap_arch >= 0 && cap_disas_monitor(&s.info, pc, nb_insn)) {

        return;

    }



    if (!s.info.print_insn) {

        monitor_printf(mon, "0x" TARGET_FMT_lx

                       ": Asm output not supported on this arch\n", pc);

        return;

    }



    for(i = 0; i < nb_insn; i++) {

	monitor_printf(mon, "0x" TARGET_FMT_lx ":  ", pc);

        count = s.info.print_insn(pc, &s.info);

	monitor_printf(mon, "\n");

	if (count < 0)

	    break;

        pc += count;

    }

}
