import pip # installing required packages requires pip installation ;-)


def install(package):
    pip.main(['install', package])


### https://stackoverflow.com/questions/48097428/how-to-check-and-install-missing-modules-in-python-at-time-of-execution
def install_required_packages(modules_to_try):
    for module in modules_to_try:
        try:
            __import__(module[0])
            print(">> " + module[0] + " already installed!")
        except ImportError as e:
            #install(e.name)
            install(module[1])


def main():
    install_pkg = True
    
    if (install_pkg):
        required_pkg_list = [('arff', 'liac-arff'),  # Library https://pypi.org/project/liac-arff/. For read/write arff files
                             ('numpy', 'numpy'),
                             ('cv2', 'opencv-python'),
                             ('tensorflow', 'tensorflow') 
                             #,
                             #('torch', 'torchvision --no-cache-dir')
                             ]
    install_required_packages(required_pkg_list)


if __name__ == "__main__":
    main()
