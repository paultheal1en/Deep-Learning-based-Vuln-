void cpu_interrupt(CPUState *s)

{

    s->interrupt_request = 1;

}
