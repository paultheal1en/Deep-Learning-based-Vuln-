qemu_irq *pxa2xx_pic_init(target_phys_addr_t base, CPUState *env)

{

    PXA2xxPICState *s;

    int iomemtype;

    qemu_irq *qi;



    s = (PXA2xxPICState *)

            qemu_mallocz(sizeof(PXA2xxPICState));

    if (!s)

        return NULL;



    s->cpu_env = env;



    s->int_pending[0] = 0;

    s->int_pending[1] = 0;

    s->int_enabled[0] = 0;

    s->int_enabled[1] = 0;

    s->is_fiq[0] = 0;

    s->is_fiq[1] = 0;



    qi = qemu_allocate_irqs(pxa2xx_pic_set_irq, s, PXA2XX_PIC_SRCS);



    /* Enable IC memory-mapped registers access.  */

    iomemtype = cpu_register_io_memory(pxa2xx_pic_readfn,

                    pxa2xx_pic_writefn, s, DEVICE_NATIVE_ENDIAN);

    cpu_register_physical_memory(base, 0x00100000, iomemtype);



    /* Enable IC coprocessor access.  */

    cpu_arm_set_cp_io(env, 6, pxa2xx_pic_cp_read, pxa2xx_pic_cp_write, s);



    register_savevm(NULL, "pxa2xx_pic", 0, 0, pxa2xx_pic_save,

                    pxa2xx_pic_load, s);



    return qi;

}
