static void sunkbd_handle_event(DeviceState *dev, QemuConsole *src,

                                InputEvent *evt)

{

    ChannelState *s = (ChannelState *)dev;

    int qcode, keycode;

    InputKeyEvent *key;



    assert(evt->type == INPUT_EVENT_KIND_KEY);

    key = evt->u.key;

    qcode = qemu_input_key_value_to_qcode(key->key);

    trace_escc_sunkbd_event_in(qcode, QKeyCode_lookup[qcode],

                               key->down);



    if (qcode == Q_KEY_CODE_CAPS_LOCK) {

        if (key->down) {

            s->caps_lock_mode ^= 1;

            if (s->caps_lock_mode == 2) {

                return; /* Drop second press */

            }

        } else {

            s->caps_lock_mode ^= 2;

            if (s->caps_lock_mode == 3) {

                return; /* Drop first release */

            }

        }

    }



    if (qcode == Q_KEY_CODE_NUM_LOCK) {

        if (key->down) {

            s->num_lock_mode ^= 1;

            if (s->num_lock_mode == 2) {

                return; /* Drop second press */

            }

        } else {

            s->num_lock_mode ^= 2;

            if (s->num_lock_mode == 3) {

                return; /* Drop first release */

            }

        }

    }



    keycode = qcode_to_keycode[qcode];

    if (!key->down) {

        keycode |= 0x80;

    }

    trace_escc_sunkbd_event_out(keycode);

    put_queue(s, keycode);

}
