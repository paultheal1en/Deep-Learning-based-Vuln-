static void set_irq_level(void *opaque, int n, int level)

{

    NVICState *s = opaque;

    VecInfo *vec;



    n += NVIC_FIRST_IRQ;



    assert(n >= NVIC_FIRST_IRQ && n < s->num_irq);



    trace_nvic_set_irq_level(n, level);



    /* The pending status of an external interrupt is

     * latched on rising edge and exception handler return.

     *

     * Pulsing the IRQ will always run the handler

     * once, and the handler will re-run until the

     * level is low when the handler completes.

     */

    vec = &s->vectors[n];

    if (level != vec->level) {

        vec->level = level;

        if (level) {

            armv7m_nvic_set_pending(s, n);

        }

    }

}
