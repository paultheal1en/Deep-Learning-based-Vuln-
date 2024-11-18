void pxa27x_register_keypad(struct pxa2xx_keypad_s *kp, struct keymap *map,

        int size)

{

    kp->map = (struct keymap *) qemu_mallocz(sizeof(struct keymap) * size);



    if(!map || size < 0x80) {

        fprintf(stderr, "%s - No PXA keypad map defined\n", __FUNCTION__);

        exit(-1);

    }



    kp->map = map;

    qemu_add_kbd_event_handler((QEMUPutKBDEvent *) pxa27x_keyboard_event, kp);

}
