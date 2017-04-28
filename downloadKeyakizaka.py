# -*- coding: utf-8 -*-
import subprocess
import sys
import argparse

def run( cmd ):
    subprocess.call(cmd,shell=True)

def googliser( phrase , destination=None , lower=1000 , number=25 , quiet=False , upper=0 , googliser="./googliser/googliser.sh" ):
    option_quiet = ""
    if  quiet :
        option_quiet="-q"
    run( "%s -g -p %s -l %d -n %d %s -u %d" % ( googliser , phrase , lower , number , option_quiet , upper ) )
    if  destination!=None :
        run("mkdir -p "+destination)
        run("mv %s/* %s/" % (phrase,destination) )
        run("rmdir "+phrase)

member = [
    ("石森","虹花","ishimori","nijika"),
    ("今泉","佑唯","imaizumi","yui"),
    ("上村","莉菜","uemura","rina"),
    ("尾関","梨香","ozeki","rika"),
    ("織田","奈那","oda","nana"),
    ("小池","美波","koike","minami"),
    ("小林","由依","kobayashi","yui"),
    ("齋藤","冬優花","saitou","fuyuka"),
    ("佐藤","詩織","satou","shiori"),
    ("志田","愛佳","shida","manaka"),
    ("菅井","友香","sugai","yuuka"),
    ("鈴本","美愉","suzumoto","miyu"),
    ("長沢","菜々香","nagasawa","nanako"),
    ("土生","瑞穂","habu","miduho"),
    ("原田","葵","harada","aoi"),
    ("平手","友梨奈","hirate","yurina"),
    ("守屋","茜","moriya","akane"),
    ("米谷","奈々未","yonetani","nanami"),
    ("渡辺","梨加","watanabe","rika"),
    ("渡邉","理佐","watanabe","risa"),
    ("長濱","ねる","nagahama","neru") ]


desc="This python script will download images of Keyakizaka46's members (Kanji Keyaki only) from Google image search. Depend on googliser.sh(github.com/teracow/googliser)."

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("dir",help="Downloaded data is stored like \"DIR/NAME/google-image-(****).jpg\".")
parser.add_argument("-n","--number",
                    metavar="N",
                    default=25,
                    type = int,
                    help="<default=25>Number of images to download. Maximum of 1000.")
parser.add_argument("--googliser",
                    metavar="GOOGLISER",
                    default="./googliser/googliser.sh",
                    help="<default=./googliser/googliser.sh>")
args = parser.parse_args()




dir = args.dir.rstrip("/")
run("mkdir -p "+dir)

for name in member:
    googliser( name[0]+name[1] , destination=dir+"/"+name[2]+name[3] , number=args.number )
