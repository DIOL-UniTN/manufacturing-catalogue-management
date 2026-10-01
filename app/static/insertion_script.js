$(document).ready(function () {
  console.log('✅ DOM pronto e script caricato');

  let selectedIt = null;
  let selectedEn = null;

  // Ripristina l’input principale
  const saved = localStorage.getItem('savedUserInput');
  if (saved) {
    console.log('📥 Ripristinato da localStorage:', saved);
    $('#userInput').val(saved);
  }

  // Funzione di setup per IT/EN
  function setupSelection(selector, setter) {
    $(selector).on('click', 'input', function () {
      const $this = $(this);
      const was = $this.hasClass('selected');
      console.log(`🖱️ Click su ${selector}:`, $this.val());

      // Deseleziona tutto
      $(selector + ' input')
        .removeClass('selected')
        .prop('readonly', true);

      if (!was) {
        $this
          .addClass('selected')
          .prop('readonly', false)
          .focus();
        setter($this);
      } else {
        setter(null);
      }
      checkReady();
    });
  }

  // Abilita/disabilita submit
  function checkReady() {
    const ready = Boolean(selectedIt && selectedEn);
    console.log('🧪 checkReady → IT:', selectedIt?.val(), 'EN:', selectedEn?.val(), '→ ready=', ready);
    $('#createBtn').prop('disabled', !ready);
  }

  // Inizializza
  // setupSelection('#descIt', it => selectedIt = it);
  // setupSelection('#descEn', en => selectedEn = en);

  function showSuggestions(it_suggestions, en_suggestions) {
    $('#descIt').empty();
    $('#descEn').empty();

    it_suggestions.forEach(element => {
      const $item = $(`
        <input type="text" class="list-group-item editable form-control mb-2" value="${element}" readonly>
      `);
      $('#descIt').append($item)
    });

    en_suggestions.forEach(element => {
      const $item = $(`
        <input type="text" class="list-group-item editable form-control mb-2" value="${element}" readonly>
      `);
      $('#descEn').append($item)
    });
  }

  function loading(){
    $('#descIt').empty();
    $('#descEn').empty();

    const $loadit = $(`
      <div class="spinner-border text-primary" role="status">
      <span class="visually-hidden">Loading...</span>
      </div>
    `);
    const $loaden = $(`
      <div class="spinner-border text-primary" role="status">
      <span class="visually-hidden">Loading...</span>
      </div>
    `);

    $('#descIt').append($loadit)
    $('#descEn').append($loaden)
  }

  $('#submitBtn').on('click', function() {
    const input = $('#userInput').val().trim();
    localStorage.setItem('savedUserInput', input);
    loading()
    if (input) {
      $.ajax({
        url: '/get_suggestions',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ input: input }),
        success: function (data) {
          const { ita, eng } = data;
          showSuggestions(ita, eng)
          console.log(data);
        },
        error: function () {
          alert('Error with the suggestion request');
        }
      });
    }
  })

  $(document).on('keydown', function(event) {
    if (event.key === 'Enter') {
      const input = $('#userInput').val().trim();
      localStorage.setItem('savedUserInput', input);
      if (input) {
        $.ajax({
          url: '/get_suggestions',
          method: 'POST',
          contentType: 'application/json',
          data: JSON.stringify({ input: input }),
          success: function (data) {
            const { ita, eng } = data;
            showSuggestions(ita, eng)
            console.log(data);
          },
          error: function () {
            alert('Error with the suggestion request');
          }
        });
      }
    }
  });
    
  // Submit unico
  $('#createBtn').on('click', function () {
    console.log('🖱️ createBtn cliccato');
    if (!(selectedIt && selectedEn)) {
      console.warn('❌ Mancano selezioni!');
      return;
    }
    // Salva input principale
    const main = $('#userInput').val().trim();
    localStorage.setItem('savedUserInput', main);
    console.log('📥 Salvato main input:', main);

    // Prepara payload
    const payload = {
      code: $('#materialCode').val().trim(),
      desc_it: selectedIt.val().trim(),
      desc_en: selectedEn.val().trim()
    };
    console.log('📤 Payload:', payload);

    // AJAX
    $.ajax({
      method: 'POST',
      url: '/submit_insertion',
      contentType: 'application/json',
      data: JSON.stringify(payload),
    })
    .done(resp => {
      console.log('✅ Response:', resp);
      alert('Inserimento OK');
      // window.location.href = '/';
    })
    .fail((xhr, status, err) => {
      console.error('❌ Errore AJAX:', status, err);
      alert('Errore invio');
    });
  });

  // Cancel
  $('#cancelBtn').on('click', function () {
    console.log('🖱️ cancelBtn cliccato');
    const main = $('#userInput').val().trim();
    localStorage.setItem('savedUserInput', main);
    console.log('📥 Salvato main input:', main);
    window.location.href = '/';
  });
});
