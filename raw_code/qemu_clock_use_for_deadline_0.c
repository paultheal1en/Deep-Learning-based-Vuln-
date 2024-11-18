bool qemu_clock_use_for_deadline(QEMUClockType type)

{

    return !(use_icount && (type == QEMU_CLOCK_VIRTUAL));

}
