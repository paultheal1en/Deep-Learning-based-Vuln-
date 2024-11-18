static void win_stdio_close(CharDriverState *chr)

{

    WinStdioCharState *stdio = chr->opaque;



    if (stdio->hInputReadyEvent != INVALID_HANDLE_VALUE) {

        CloseHandle(stdio->hInputReadyEvent);

    }

    if (stdio->hInputDoneEvent != INVALID_HANDLE_VALUE) {

        CloseHandle(stdio->hInputDoneEvent);

    }

    if (stdio->hInputThread != INVALID_HANDLE_VALUE) {

        TerminateThread(stdio->hInputThread, 0);

    }



    g_free(chr->opaque);

    g_free(chr);

}
