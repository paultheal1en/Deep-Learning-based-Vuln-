static void vmmouse_reset(DeviceState *d)
{
    VMMouseState *s = container_of(d, VMMouseState, dev.qdev);
    s->status = 0xffff;
    s->queue_size = VMMOUSE_QUEUE_SIZE;
}