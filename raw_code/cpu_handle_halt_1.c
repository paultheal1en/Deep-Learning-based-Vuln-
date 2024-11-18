static inline bool cpu_handle_halt(CPUState *cpu)

{

    if (cpu->halted) {

#if defined(TARGET_I386) && !defined(CONFIG_USER_ONLY)

        if ((cpu->interrupt_request & CPU_INTERRUPT_POLL)

            && replay_interrupt()) {

            X86CPU *x86_cpu = X86_CPU(cpu);

            qemu_mutex_lock_iothread();

            apic_poll_irq(x86_cpu->apic_state);

            cpu_reset_interrupt(cpu, CPU_INTERRUPT_POLL);

            qemu_mutex_unlock_iothread();

        }

#endif

        if (!cpu_has_work(cpu)) {

            current_cpu = NULL;

            return true;

        }



        cpu->halted = 0;

    }



    return false;

}
