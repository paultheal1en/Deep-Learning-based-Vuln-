RTCState *rtc_init(int base_year)

{

    ISADevice *dev;



    dev = isa_create("mc146818rtc");

    qdev_prop_set_int32(&dev->qdev, "base_year", base_year);

    qdev_init(&dev->qdev);

    return DO_UPCAST(RTCState, dev, dev);

}
