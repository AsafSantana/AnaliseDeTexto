# coding: utf-8
import nltk

#nltk.download()


#print(base[1])


basetreinamento = [
('febre alta > 38.5ºC','Dengue'), 
('dores musculares intensas','Dengue'),
('dor ao movimentar os olhos','Dengue'),
('mal estar','Dengue'),
('falta de apetite','Dengue'),
('dor de cabeça','Dengue'),
('manchas vermelhas no corpo.','Dengue'),
('dor abdominal intensa e contínua','Dengue'),
('vômitos persistentes','Dengue'),
('acumulação de líquidos ascites, derrame pleural, derrame pericárdico','Dengue'),
('sangramento de mucosa ou outra hemorragia','Dengue'),
('aumento progressivo do hematócrito','Dengue'),
('queda abrupta das plaquetas','Dengue'),
('tosse crônica','câncer de pulmão'),
('tosse com sange','câncer de pulmão'),
('chiado','câncer de pulmão'),
('falta de ar','câncer de pulmão'),
('rouquidão','câncer de pulmão'),
('infecções pulmonares como pneumonia','câncer de pulmão')
]





# palavras ja cadastradas no nltk - uso as do idioma portugues
stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
stopwordsnltk.append('vou')
stopwordsnltk.append('tão') # caso eu queira adicionar mais stopwords no banco de dados 
#print(stopwordsnltk)


# Remove as stopwords dos dados passados
def removestopwords(texto):  
    frases = []
    for (palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwordsnltk]
        frases.append((semstop, emocao))
    return frases

#print(removestopwords(base))

# retira os radicais das palavras, deixando mais leve, estou usando RSLPStemmer()
def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasessstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasessstemming.append((comstemming, emocao))
    return frasessstemming

frasescomstemmingtreinamento = aplicastemmer(basetreinamento)
#frasescomstemmingteste = aplicastemmer(baseteste)
#print(frasescomstemming)


# Busca todas as palavras da base de dados
def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

palavrastreinamento = buscapalavras(frasescomstemmingtreinamento)
#palavrasteste = buscapalavras(frasescomstemmingteste)
#print(palavras)


# mostro a frequencia que essa palavra aparece na base de dados
def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequenciatreinamento = buscafrequencia(palavrastreinamento)
#frequenciateste = buscafrequencia(palavrasteste)
#print(frequencia.most_common(50))


# Mostra palavras que aparem uma unica vez usando a função .keys()
def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

palavrasunicastreinamento = buscapalavrasunicas(frequenciatreinamento)
#palavrasunicasteste = buscapalavrasunicas(frequenciateste)
#print(palavrasunicastreinamento)

#print(palavrasunicas)

# Recebe um documento e mostra se essa palavra existe ou não no documento
def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicastreinamento:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

#caracteristicasfrase = extratorpalavras(['am', 'nov', 'dia'])
#print(caracteristicasfrase)




basecompletatreinamento = nltk.classify.apply_features(extratorpalavras, frasescomstemmingtreinamento)
#basecompletateste = nltk.classify.apply_features(extratorpalavras, frasescomstemmingteste)
#print(basecompleta[15])



classificador = nltk.NaiveBayesClassifier.train(basecompletatreinamento)


teste = 'Em contraste, o tipo mais comum de câncer de pulmão em mulheres é o adenocarcinoma, que se desenvolve nas regiões externas dos pulmões. Esses tumores podem crescer bastante ou se espalhar antes de provocar qualquer sintoma. Os sintomas iniciais podem incluir falta de ar, fadiga e dores nas costas e no ombro, tosse crônica, tosse com sangue, chiado no pulmão e falta de ar'
#teste = 'No entanto, a infecção por pode ser assintomática (sem sintomas), leve ou grave. Neste último caso pode levar até a morte. Normalmente, a primeira manifestação da dengue é a febre alta (39° a 40°C), de início abrupto, que geralmente dura de 2 a 7 dias, acompanhada de dor de cabeça, dores no corpo e articulações, além de prostração, fraqueza, dor atrás dos olhos, erupção e coceira na pele. Perda de peso, náuseas e vômitos são comuns. Em alguns casos também apresenta manchas vermelhas na pele. Na fase febril inicial da dengue, pode ser difícil diferenciá-la. A forma grave da doença inclui dor abdominal intensa e contínua, vômitos persistentes e sangramento de mucosas. Ao apresentar os sintomas, é importante procurar um serviço de saúde para diagnóstico e tratamento adequados, todos oferecidos de forma integral e gratuita por meio do Sistema Único de Saúde (SUS).'

testestemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavrastreinamento) in teste.split():
    comstem = [p for p in palavrastreinamento.split()]
    testestemming.append(str(stemmer.stem(comstem[0])))
#print(testestemming)

novo = extratorpalavras(testestemming)
#print(novo)

print()
print()
print('------ Texto de Entada ----------')
print()
print(teste)

print()
print('------ Doença com mais probabilidade - Algoritmo Naive Bayes ----------')
print()
print(classificador.classify(novo))
print()

print('------ Porcentagem das doenças que foram treinadas----------')
print()
distribuicao = classificador.prob_classify(novo)
for classe in distribuicao.samples():
   print("%s: %f" % (classe, distribuicao.prob(classe)*100))










