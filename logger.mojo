
from python import Python


struct Logger:
    @staticmethod
    fn warn(message: String) raises:
        var termcolor = Python.import_module("termcolor")
        var sys = Python.import_module("sys")
        var colored = termcolor.colored
        var color_text = colored("Notcie: ", "yellow", None, ["bold"])
        print(color_text, message)

    @staticmethod
    fn notice(message: String) raises:
        var termcolor = Python.import_module("termcolor")
        var sys = Python.import_module("sys")
        var colored = termcolor.colored
        var color_text = colored("Notice: ", "green", None, ["bold"])
        print(color_text, message)

    @staticmethod
    fn error(message: String) raises:
        var termcolor = Python.import_module("termcolor")
        var sys = Python.import_module("sys")
        var colored = termcolor.colored
        var color_text = colored("Error: ", "red", None, ["bold"])
        print(color_text, message)

    @staticmethod
    fn status(message: String) raises:
        var termcolor = Python.import_module("termcolor")
        var sys = Python.import_module("sys")
        var colored = termcolor.colored
        var color_text = colored("Status: ", "white", None, ["bold"])
        print(color_text, message)

    @staticmethod
    fn cls() raises:
        var subprocess = Python.import_module("subprocess")
        var os = Python.import_module("os")
        var call = subprocess.call('clear' if os.name == 'posix' else 'cls')