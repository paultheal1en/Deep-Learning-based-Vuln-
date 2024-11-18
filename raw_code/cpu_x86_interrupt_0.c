void cpu_x86_interrupt(CPUX86State *s)

{

    s->interrupt_request = 1;

}
