static bool cpu_thread_is_idle(CPUState *cpu)

{

    if (cpu->stop || cpu->queued_work_first) {

        return false;

    }

    if (cpu->stopped || !runstate_is_running()) {

        return true;

    }

    if (!cpu->halted || qemu_cpu_has_work(cpu) ||

        kvm_async_interrupts_enabled()) {

        return false;

    }

    return true;

}
