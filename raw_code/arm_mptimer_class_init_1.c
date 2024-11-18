static void arm_mptimer_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = arm_mptimer_realize;

    dc->vmsd = &vmstate_arm_mptimer;

    dc->reset = arm_mptimer_reset;

    dc->no_user = 1;

    dc->props = arm_mptimer_properties;

}
