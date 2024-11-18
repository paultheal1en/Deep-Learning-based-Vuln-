void qemu_del_wait_object(HANDLE handle, WaitObjectFunc *func, void *opaque)

{

    int i, found;

    WaitObjects *w = &wait_objects;



    found = 0;

    for (i = 0; i < w->num; i++) {

        if (w->events[i] == handle)

            found = 1;

        if (found) {

            w->events[i] = w->events[i + 1];

            w->func[i] = w->func[i + 1];

            w->opaque[i] = w->opaque[i + 1];

        }

    }

    if (found)

        w->num--;

}
