
from python import Python
import time

struct Logger:
    var filename: String
    var file: FileHandle

    fn __init__(inout self, filepath: String) raises:
        var now = Logger.now()
        self.filename = filepath+"/log-"+str(now).replace(" ", "-")+".txt"
        self.file = open(self.filename, 'w')
        self.file.write("Log session created at "+str(now)+"\n")

    fn __del__(owned self):
        try:
            self.file.write("Log session ended at "+str(Logger.now())+"\n")
            self.file.close()
        except:
            pass

    fn warn(inout self, message: String) raises:
        var termcolor = Python.import_module("termcolor")
        var colored = termcolor.colored
        var date = colored(str(Logger.now()), color="grey", on_color="on_grey")
        print(colored("Warning", color="grey", on_color="on_yellow"), date, message)
        self.file.write("[Warning] " + "["+str(Logger.now())+"] " + message + "\n")

    fn notice(inout self, message: String) raises:
        var termcolor = Python.import_module("termcolor")
        var colored = termcolor.colored
        var date = colored(str(Logger.now()), color="grey", on_color="on_grey")
        print(colored("Notice", color="grey", on_color="on_green"), date, message)
        self.file.write("[Notice] " + "["+str(Logger.now())+"] " + message + "\n")
    
    fn error(inout self, error: Error) raises:
        var termcolor = Python.import_module("termcolor")
        var colored = termcolor.colored
        var date = colored(str(Logger.now()), color="grey", on_color="on_grey")
        print(colored("Error", color="grey", on_color="on_red"), date, error)
        self.file.write("[Error] " + "["+str(Logger.now())+"]\nTraceback:\n" + error + "\n")

    
    fn status(inout self, message: String) raises:
        var termcolor = Python.import_module("termcolor")
        var colored = termcolor.colored
        var date = colored(str(Logger.now()), color="grey", on_color="on_grey")
        print(colored("Status", color="grey", on_color="on_grey"), date, message)
        self.file.write("[Status] " + "["+str(Logger.now())+"] " + message + "\n")


    @staticmethod
    fn cls() raises:
        var subprocess = Python.import_module("subprocess")
        var os = Python.import_module("os")
        var call = subprocess.call('clear' if os.name == 'posix' else 'cls')

    @staticmethod
    fn now() raises -> PythonObject:
        var datetime = Python.import_module("datetime")
        var now = datetime.datetime.now()
        return now

fn main() raises:
    var logger = Logger("logs")
    logger.status("Test status")
    logger.warn("Test status")
    logger.notice("Test status")