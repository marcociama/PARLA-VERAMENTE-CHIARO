import json
import os
import re
import time
import openai

# === CONFIGURATION ===
FOLDER_PATH = "./data"  # Use relative folder in repo
INPUT_JSON = os.path.join(FOLDER_PATH, "sentences.json")
OUTPUT_JSON = os.path.join(FOLDER_PATH, "transcriptions.json")
LANGUAGE = "it"  # Italian

# === API KEY ===
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set your OpenAI API key in the OPENAI_API_KEY environment variable.")

lines = """E chesto capisce tu: 'e denare!
E cu' 'e denare t'he accattato tutto chello ca he voluto!
Ma Filumena Marturano ha fatto correre essa a te! E currive senza ca te n'addunave.
"E ancora he 'a correre, ancora he 'a iettà 'o sango a capi comme se campa e se prucede 'a galantomo!"
O miédeco nun sapeva niente.
"Ce ha creduto pur'isso, e ce avev' 'a credere!"
"Qualunque femmena, doppo vinticinc'anne che ha passato vicino a te, se mette in agonia."
T'aggio fatto 'a serva!
"A serva ll'aggio fatta pè vinticinc'anne, e vuie 'o ssapite."
"E maie ca t'avesse visto sottomessa, che ssaccio?"
E avev' 'a chiagnere pe' te?
Era troppo bello 'o mobile.
"E quanno me vulive vedé 'e durmi, tu?"
A strada d' 'a casa t' 'a scurdave.
"E mmeglie feste, 'e meglie Natale me ll'aggio passate sola comm' a na cana."
Saie quanno se chiagne?
Quanno se cunosce 'o bbene e nun se pò avé!
Ma Filumena Marturano bene nun ne cunosce... e quanno se cunosce sulo 'o mmale nun se chiagne.
"A suddisfazione 'e chiagnere, Filumena Marturano, nun l'ha pututa maie avé!"
"Comm' all'ultima femmena m' he trattato, sempe!"
"Ma mo, all'urdemo all'urdemo, a cinquantaduie anne, se retira cu' 'e fazzulette spuorche 'e russetto, ca me fanno schifo."
"A chella chi? .,. A chella chi?"
Appriesso a chella schifosa!
Che te cride ca nun l'avevo capito?
"Tu buscie nun ne saie dicere, e chisto è 'o difetto tuio."
"Cinquantaduie anne, e se permette 'e se mettere cu' na figliola 'e vintiduie!"
Nun se ne mette scuorno
"E mm' 'a mette dint' 'a casa, dicenno ca era l'infermiera... Pecché isso se credeva overo ca io stevo murenno."
Madonna... quanto me faie schifo!
"E se io stevo murenno overamente, tu chesto avisse fatto?"
"Ma pecché, tu murive e io nun avev' 'a magnà cchiu?"
Nun m'avev' 'a sustené?
Ch'e rrose mmiez' 'a tavula?
"Ma pecché, nun ero padrone d' 'e mmettere?"
Quanto me faie ridere!
"Ma che me ne mporta 'e te, d' 'a figliola che t'ha fatto perdere 'a capa, 'e tutto chello ca me dice?"
Ma tu te cride overo ca io ll'aggio fatto pe' te?
"Ma io nun te curo, nun t'aggio maie curato."
"Na femmena comm' a mme, ll'he ditto tu e mm' 'o stai dicenno 'a vinticinc'anne, se fa 'e cunte."
"Me sierve... Tu, me sierve!"
E denare! E nun te l'avarria date?
"Filume', tu afforza me vuo pògnere?"
"Ma pecché nun avev' 'a mangià, secondo voi?"
Qua sta 'a cena.
"Quanno site venuto ogge p'urdinà 'a cena, ve ricurdate?"
"Cient’anne arreto ch’era viva Vava,"
"nnante che ffosse Vartommeo Coglione,"
dicea no cierto che l’auciello arava
a ttiempo che sguigliaje lo Sciatamone.
"Nc’era lo Rre Marruocco che s’armava, "
"panzera, lanza longa e toracone, "
e po’ jeva a ttrovà li Mammalucche
"co balestre, spigarde, e co ttrabucche. "
"Chillo fu tiempo che Berta filava, "
co chillo doce vivere a l’antica!
"Portave brache, e nullo delleggiava! "
"Ogn’anno, il due novembre, c’è l’usanza per i defunti andare al Cimitero,"
"Si pe la via na femmena passava, "
le dicevano: “Ddio la benedica!”.
"Mo, s’uno parla, e chella se corruzza. "
Chi te pienze che ssia? Monna Maruzza.
"O bell’ausanza, e ddove si’ squagliata? "
"Pecchè non tuorne, o doce tiempo antico? "
"Pigliave co lo bisco, a na chiammata, "
cient’aucelluzze a no trunco de fico!
"Le ffemmene, addorose de colata, "
"Ma chi te cride d’essere, nu ddio? Cca dinto, ‘o vvuò capì ca simmo eguae?"
"danzanno tutte ‘n chietta, (oh bona fede!)"
"Madonna, si ce penzo che paura! ma po’ facett’ un’anema ‘e curaggio"
Dove se trova mai tanta lianza!
"Lo marito sì ccaro a la mogliera, "
che a mano a mano ‘ntravano a na danza
co chella ciaramella tant’allera!
"Vedive, a chioppa a chioppa, na paranza "
co chell’antica e semprece manera!
"Lo viecchio a chillo tiempo era zitiello, "
co le brache stringate e ‘n jopponciello.
Chillo non era tiempo ammagagnato!
"Le ffemmene assettate mmiezo chiazza, "
"non c’era n’ommo ch’avesse parlato, "
ca vernava ‘n cajola la cajazza.
"Chill’ommo, che ‘n chill’anno era nzorato, "
era tenuto pe gallo de razza.
Ll’uno co ll’autro lo mostrav’a dito:
"“Chillo che passa mo, chill’è lo zito!” "
Tutte le bon’ausanze so’ lassate!
Le rose mo deventano papagne!
"Lo vicenato, ‘n chietta e ‘n lebertate, "
"a chillo tiempo jevano a li vagne, "
Pecché ‘ncopp’a sta terra  femmene comme a te  non ce hanna sta pé n’ommo  onesto comme a me!…
Si ddoce comme ‘o zucchero
"E ghievano abbracciate a otto, a diece, "
cchiù ghianche e rosse che le mmela-diece
Chella co la gonnella de scarlata
portava perne grosse comm’antrita.
O Puorto 'e Napule è nu puorto mpizzato int''o Gurfo 'e Napule ca se spanne int''a custiera d''a cità 'e Napule.
È uno d''e cchiù mpurtanti puorte d'Europa.​
Melito 'e Napule (ditto Mêlito d''a ggente) è nu comune 'e 38.062 crestiane d''a pruvincia 'e Napule.
"O Regno 'e Napule (nomme ufficiale: Regno 'e Sicilia citeriore) è 'o nomme cu cui è canusciuto nu stato indipendente, ca esisteva tra 'o XIII e 'o XIX seculo e ca currispunneva a ll'attuale reggione 'e ll'Italia meridiunale, 'ncluse Abruzzo e parte d""o Lazzio, ma cca lassava fore 'a Sicilia"
"O gurfo 'e Napule è na 'nzenatura d''o Mar Tirreno meriddionale, cumprèsce 'nfra 'a penisula flegrea a nord-ovest (capo Miseno) e 'a penisula surrientina a sud-est (punta Campanella)"
A Zona Nnustriale (o Gianturco) è nu rione 'e 6082 crestiani d''a part' 'e levante 'e Napule.
O nomme Gianturco vene d''a via prencepale d''o rione: via Emanuele Gianturco.​
Tutte 'e cristiane nasceno libbere e pare pe degnetà e jusse; teneno cereviello e cuscienza e hanno 'a faticà ll'une cu ll'ate cu nu spireto 'e fraternetà.​
"O nnapulitano è na lengua rumanza ca se parla 'n Campania e all'ate parte d""o Sud-Italia."
"Soletamente, però, so' cchiammate accussí tutt' 'e dialette ca se parlano ô Sud cuntinentale (Campania, Abruzzo, Lucania, nord d'a Calavrea, nord d'a Puglia, Molise, sudd d""o Lazzio e na pparte d""e Marche) e parlate ca se ntenneno ll'une cu ll'ate e teneno cierte rréole símmele."
"Chesta idea nun è 'a stessa pe tutte 'e lenguiste; cierte diceno mméce ca, cu tutte 'e ddifferenze ca ce stanno nfra chisti dialette e p' 'o fatto ca nun ce sta nisciuna standardizzazione, nun se pô parlà 'e na ""lengua napulitana"" e parlano mméce 'e nu cuntinuo 'e ""dialette taliane meridiunale ntermedie""."
O Nnapulitano è na lengua o nu dialetto?
Chi 'o pparla e addò?​
"E ancora, quanno s'avess' 'a parlà Taliano e quanno 'o dialetto?​"
Nun ce sta periculo ca 'stu fatto succede a Napule.​
"O fatto è ca ê ffamiglie e ê putecare, pe Napule e 'e pizze vicine, lle piace 'e cchiù 'e parlà Nnapulitano, 'a lengua antica d''o Rregno d''e Ddoje Secilie."
"Museciste e judece, chianchiere e barriste sciuliéano assaje spisso int''o ddialetto lucale."
"Pe tutte ll'anne ca io songo stata a Napule, aggio cercato 'e tutt' 'e mmanere, ma quase sempe a vvacante, comm'avevo 'a fa' pe trasì int' 'a suggità secreta d''a ggente che parlano 'o Nnapulitano."
Ogni scarrafone è bello 'a mamma soja.​
A cucina piccerella fa la casa granne.
Ccà nisciuno è fesso.​
Chi nun sape chiagnere nun sape manco rirere
"Vide Napule, e po' muore.​"
A 'o core nun se cumanna.
Tutto 'o lassato è perduto.
"Chiù 'a capa è vacante, chiù 'a lengua è longa."
"Si sí felice, tienetello pe te; ca addó nun arriva l’invidia, arriva 'a sfurtuna."
"Vulesse, potesse e facesse, ereno tre fesse.​"
Nun ve 'ntricate tra marito e mugliera.​
"Chi bella vo' paré', pene e guaie adda' paté'."
Se mangiaie duie chile d'uva 'e nascosto d' 'a mamma.
"Già, me faceva male 'o pede."
E pecché avess' 'a ji' malamente?
Nu mumento... Io nun me so' sbagliata.
Ma a me mi pare che si abusi della cortesia altrui!
"Guè, ccà se tratta 'e na cosa seria, t'aggio ditto!"
"Nun tene 'o curaggio d'ascì ccà fore. Iammo nnanze, avvoca'."
"Statte zitto, ca manco tu he capito."
Io nun saccio leggere e po' carte nun n'accetto!
"Allora io aggio spiso na vita pè furmà na famiglia, e 'a legge nun m' 'o permette?"
"E chi si' tu, ca me vuò mpedì 'e dicere, vicin' 'e figlie mieie, ca me so' ffiglie?"
"E va bbuono, mo basta!"
Dimane me manno a piglià 'a rrobba mia.
"Te putevo dicere ca tutt'e tre t'erano figlie, ce avarrisse creduto... T' 'o ffacevo credere!"
"No una, ma ciento vote, me l'avarrisse fatto accidere!"
"Me mettette appaura 'e t' 'o ddicere! Sulo per me, è vivo 'o figlio tuio!"
Hann’ ‘a essere eguale tutt' 'e tre!"""

output = []
for idx, line in enumerate(lines.strip().split("\n"), start=2):
    output.append({
        "id": f"{idx:03d}",         # "002", "003", … up to "142"
        "neapolitan": line.strip()
    })

with open(INPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)