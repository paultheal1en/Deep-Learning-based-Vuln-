CoroutineAction qemu_coroutine_switch(Coroutine *from_, Coroutine *to_,

                                      CoroutineAction action)

{

    CoroutineWin32 *from = DO_UPCAST(CoroutineWin32, base, from_);

    CoroutineWin32 *to = DO_UPCAST(CoroutineWin32, base, to_);



    current = to_;



    to->action = action;

    SwitchToFiber(to->fiber);

    return from->action;

}
