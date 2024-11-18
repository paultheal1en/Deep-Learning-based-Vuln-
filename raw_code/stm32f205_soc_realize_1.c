static void stm32f205_soc_realize(DeviceState *dev_soc, Error **errp)

{

    STM32F205State *s = STM32F205_SOC(dev_soc);

    DeviceState *syscfgdev, *usartdev, *timerdev;

    SysBusDevice *syscfgbusdev, *usartbusdev, *timerbusdev;

    qemu_irq *pic;

    Error *err = NULL;

    int i;



    MemoryRegion *system_memory = get_system_memory();

    MemoryRegion *sram = g_new(MemoryRegion, 1);

    MemoryRegion *flash = g_new(MemoryRegion, 1);

    MemoryRegion *flash_alias = g_new(MemoryRegion, 1);



    memory_region_init_ram(flash, NULL, "STM32F205.flash", FLASH_SIZE,

                           &error_abort);

    memory_region_init_alias(flash_alias, NULL, "STM32F205.flash.alias",

                             flash, 0, FLASH_SIZE);



    vmstate_register_ram_global(flash);



    memory_region_set_readonly(flash, true);

    memory_region_set_readonly(flash_alias, true);



    memory_region_add_subregion(system_memory, FLASH_BASE_ADDRESS, flash);

    memory_region_add_subregion(system_memory, 0, flash_alias);



    memory_region_init_ram(sram, NULL, "STM32F205.sram", SRAM_SIZE,

                           &error_abort);

    vmstate_register_ram_global(sram);

    memory_region_add_subregion(system_memory, SRAM_BASE_ADDRESS, sram);



    pic = armv7m_init(get_system_memory(), FLASH_SIZE, 96,

                      s->kernel_filename, s->cpu_model);



    /* System configuration controller */

    syscfgdev = DEVICE(&s->syscfg);

    object_property_set_bool(OBJECT(&s->syscfg), true, "realized", &err);

    if (err != NULL) {

        error_propagate(errp, err);

        return;

    }

    syscfgbusdev = SYS_BUS_DEVICE(syscfgdev);

    sysbus_mmio_map(syscfgbusdev, 0, 0x40013800);

    sysbus_connect_irq(syscfgbusdev, 0, pic[71]);



    /* Attach UART (uses USART registers) and USART controllers */

    for (i = 0; i < STM_NUM_USARTS; i++) {

        usartdev = DEVICE(&(s->usart[i]));

        object_property_set_bool(OBJECT(&s->usart[i]), true, "realized", &err);

        if (err != NULL) {

            error_propagate(errp, err);

            return;

        }

        usartbusdev = SYS_BUS_DEVICE(usartdev);

        sysbus_mmio_map(usartbusdev, 0, usart_addr[i]);

        sysbus_connect_irq(usartbusdev, 0, pic[usart_irq[i]]);

    }



    /* Timer 2 to 5 */

    for (i = 0; i < STM_NUM_TIMERS; i++) {

        timerdev = DEVICE(&(s->timer[i]));

        qdev_prop_set_uint64(timerdev, "clock-frequency", 1000000000);

        object_property_set_bool(OBJECT(&s->timer[i]), true, "realized", &err);

        if (err != NULL) {

            error_propagate(errp, err);

            return;

        }

        timerbusdev = SYS_BUS_DEVICE(timerdev);

        sysbus_mmio_map(timerbusdev, 0, timer_addr[i]);

        sysbus_connect_irq(timerbusdev, 0, pic[timer_irq[i]]);

    }

}
