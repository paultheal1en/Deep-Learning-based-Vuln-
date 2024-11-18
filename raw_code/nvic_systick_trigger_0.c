static void nvic_systick_trigger(void *opaque, int n, int level)

{

    NVICState *s = opaque;



    if (level) {

        /* SysTick just asked us to pend its exception.

         * (This is different from an external interrupt line's

         * behaviour.)

         */

        armv7m_nvic_set_pending(s, ARMV7M_EXCP_SYSTICK);

    }

}
