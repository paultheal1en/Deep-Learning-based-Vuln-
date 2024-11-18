static void audio_init (PCIBus *pci_bus)

{

    struct soundhw *c;

    int audio_enabled = 0;



    for (c = soundhw; !audio_enabled && c->name; ++c) {

        audio_enabled = c->enabled;

    }



    if (audio_enabled) {

        AudioState *s;



        s = AUD_init ();

        if (s) {

            for (c = soundhw; c->name; ++c) {

                if (c->enabled)

                    c->init.init_pci (pci_bus, s);

            }

        }

    }

}
