static void pxa2xx_gpio_handler_update(PXA2xxGPIOInfo *s) {

    uint32_t level, diff;

    int i, bit, line;

    for (i = 0; i < PXA2XX_GPIO_BANKS; i ++) {

        level = s->olevel[i] & s->dir[i];



        for (diff = s->prev_level[i] ^ level; diff; diff ^= 1 << bit) {

            bit = ffs(diff) - 1;

            line = bit + 32 * i;

            qemu_set_irq(s->handler[line], (level >> bit) & 1);

        }



        s->prev_level[i] = level;

    }

}
